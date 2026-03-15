import logging
import re
from collections import deque

from apted import APTED, Config
from apted.helpers import Tree
from bs4 import BeautifulSoup
from rapidfuzz.distance import Levenshtein

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text


def normalized_edit_similarity(text1: str, text2: str) -> float:
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)
    if not text1 and not text2:
        return 1.0
    return Levenshtein.normalized_similarity(text1, text2)


def extract_html_tables(markdown_text: str) -> list[str]:
    soup = BeautifulSoup(markdown_text or "", "html.parser")
    return [str(table) for table in soup.find_all("table")]


def remove_html_tables(markdown_text: str) -> str:
    soup = BeautifulSoup(markdown_text or "", "html.parser")
    for table in soup.find_all("table"):
        table.decompose()
    return soup.get_text("\n")


class TableTree(Tree):
    def __init__(self, tag, name, colspan=None, rowspan=None, content=None, *children):
        super().__init__(name, *children)
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)


class TableEditConfig(Config):
    @staticmethod
    def maximum(*sequences):
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        return float(Levenshtein.distance(*sequences)) / max(1, self.maximum(*sequences))

    def rename(self, node1, node2):
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.0
        if node1.tag == "td" and (node1.content or node2.content):
            return self.normalized_distance(node1.content, node2.content)
        return 0.0


class TEDS:
    def __init__(self, structure_only: bool = False, ignore_nodes: list[str] | None = None):
        self.structure_only = structure_only
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        self.__tokens__.append(f"<{node.tag}>")
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for child in node.getchildren():
            self.tokenize(child)
        if node.tag != "unk":
            self.__tokens__.append(f"</{node.tag}>")
        if node.tag != "td" and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                cell,
                *deque(),
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for child in node.getchildren():
                self.load_html_tree(child, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred: str, gt: str) -> float:
        from lxml import etree, html

        if not pred or not gt:
            return 0.0

        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        pred_node = html.fromstring(pred, parser=parser)
        gt_node = html.fromstring(gt, parser=parser)
        pred_tables = pred_node.xpath("//table")
        gt_tables = gt_node.xpath("//table")
        if not pred_tables or not gt_tables:
            return 0.0

        pred_table = pred_tables[0]
        gt_table = gt_tables[0]
        if self.ignore_nodes:
            etree.strip_tags(pred_table, *self.ignore_nodes)
            etree.strip_tags(gt_table, *self.ignore_nodes)

        node_count = max(len(pred_table.xpath(".//*")), len(gt_table.xpath(".//*")))
        if node_count == 0:
            return 1.0

        tree_pred = self.load_html_tree(pred_table)
        tree_gt = self.load_html_tree(gt_table)
        distance = APTED(tree_pred, tree_gt, TableEditConfig()).compute_edit_distance()
        return max(0.0, 1.0 - float(distance) / node_count)


teds = TEDS()


def compute_markdown_table_reward(pred_markdown: str, gt_markdown: str) -> float:
    pred_tables = extract_html_tables(pred_markdown)
    gt_tables = extract_html_tables(gt_markdown)

    if len(pred_tables) != len(gt_tables):
        return normalized_edit_similarity(pred_markdown, gt_markdown)

    text_reward = normalized_edit_similarity(remove_html_tables(pred_markdown), remove_html_tables(gt_markdown))
    if not gt_tables:
        return text_reward

    table_rewards = []
    for pred_table, gt_table in zip(pred_tables, gt_tables, strict=False):
        try:
            table_rewards.append(teds.evaluate(pred_table, gt_table))
        except Exception as exc:
            logger.warning("TEDS evaluation failed, fallback to table-level edit similarity: %s", exc)
            table_rewards.append(normalized_edit_similarity(pred_table, gt_table))

    table_reward = sum(table_rewards) / len(table_rewards)
    return (table_reward + text_reward) / 2


async def markdown_table_reward(args, sample: Sample, **kwargs) -> float:
    return compute_markdown_table_reward(sample.response, sample.label or "")
