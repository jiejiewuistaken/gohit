#!/usr/bin/env python3
"""Train a multi-class decision tree with PySpark and export a Graphviz image."""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Iterator, List, Optional

from graphviz import Digraph

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassificationModel, DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession


def load_iris_dataset(spark: SparkSession) -> DataFrame:
    """Load the iris dataset and return it as a Spark DataFrame."""

    try:
        from sklearn.datasets import load_iris  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "scikit-learn is required to load the iris dataset. Install it with `pip install scikit-learn`."
        ) from exc

    iris = load_iris(as_frame=True)
    pdf = iris.frame.rename(columns={"target": "label_str"})
    return spark.createDataFrame(pdf)


def build_pipeline(df: DataFrame) -> Pipeline:
    """Create a pipeline that indexes labels, assembles features and fits a decision tree."""

    feature_columns = [col for col in df.columns if col != "label_str"]

    stages = [
        StringIndexer(inputCol="label_str", outputCol="label", handleInvalid="keep"),
        VectorAssembler(inputCols=feature_columns, outputCol="features"),
        DecisionTreeClassifier(
            labelCol="label",
            featuresCol="features",
            maxDepth=4,
            impurity="gini",
            seed=42,
        ),
    ]

    return Pipeline(stages=stages)


def prettify_condition(text: str, feature_names: List[str]) -> str:
    pattern = re.compile(r"feature (\d+)")

    def _replace(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if 0 <= idx < len(feature_names):
            return feature_names[idx]
        return f"feature {idx}"

    return pattern.sub(_replace, text)


def parse_probabilities(probabilities: str, labels: List[str]) -> str:
    values = [v.strip() for v in probabilities.strip("[]").split(",") if v.strip()]
    try:
        numbers = [float(v) for v in values]
    except ValueError:
        return probabilities

    if len(numbers) != len(labels):
        return probabilities

    lines = [f"{label}: {prob:.3f}" for label, prob in zip(labels, numbers)]
    return "\n".join(lines)


class TreeNode:
    def __init__(
        self,
        *,
        node_type: str,
        condition: Optional[str] = None,
        else_condition: Optional[str] = None,
        prediction: Optional[str] = None,
        raw_prediction: Optional[str] = None,
        probabilities: Optional[str] = None,
        true_child: Optional["TreeNode"] = None,
        false_child: Optional["TreeNode"] = None,
    ) -> None:
        self.node_type = node_type
        self.condition = condition
        self.else_condition = else_condition
        self.prediction = prediction
        self.raw_prediction = raw_prediction
        self.probabilities = probabilities
        self.true_child = true_child
        self.false_child = false_child


def parse_debug_tree(
    debug_string: str, feature_names: List[str], labels: List[str]
) -> Optional[TreeNode]:
    lines = [line for line in debug_string.splitlines() if line.strip()]
    if not lines:
        return None

    if lines[0].startswith("DecisionTree"):
        lines = lines[1:]

    index = 0

    def indentation(text: str) -> int:
        return len(text) - len(text.lstrip(" "))

    def parse_subtree(expected_indent: int) -> Optional[TreeNode]:
        nonlocal index
        while index < len(lines):
            raw = lines[index]
            indent = indentation(raw)
            if indent < expected_indent:
                return None
            if indent > expected_indent:
                raise ValueError(
                    f"Unexpected indent {indent} at line '{raw.strip()}', expected {expected_indent}."
                )

            content = raw.strip()

            if content.startswith("If ("):
                condition = prettify_condition(content[3:-1].strip(), feature_names)
                index += 1
                true_child = parse_subtree(expected_indent + 2)

                else_condition = None
                false_child = None
                if index < len(lines):
                    next_raw = lines[index]
                    next_indent = indentation(next_raw)
                    if next_indent == expected_indent and next_raw.strip().startswith("Else"):
                        else_condition = prettify_condition(next_raw.strip()[5:-1].strip(), feature_names)
                        index += 1
                        false_child = parse_subtree(expected_indent + 2)

                return TreeNode(
                    node_type="decision",
                    condition=condition,
                    else_condition=else_condition,
                    true_child=true_child,
                    false_child=false_child,
                )

            if content.startswith("Predict:"):
                prediction_value = content.split("Predict:", 1)[1].strip()
                index += 1

                probabilities = None
                if index < len(lines):
                    maybe_prob = lines[index]
                    if indentation(maybe_prob) == expected_indent and maybe_prob.strip().startswith("Probabilities:"):
                        probabilities = maybe_prob.strip().split("Probabilities:", 1)[1].strip()
                        index += 1

                try:
                    label_idx = int(float(prediction_value))
                    label_name = labels[label_idx]
                except (ValueError, IndexError):
                    label_name = prediction_value

                prob_text = parse_probabilities(probabilities, labels) if probabilities else None

                return TreeNode(
                    node_type="leaf",
                    prediction=label_name,
                    raw_prediction=prediction_value,
                    probabilities=prob_text,
                )

            # Unrecognised line; skip and continue.
            index += 1

        return None

    return parse_subtree(expected_indent=2)


def add_nodes_to_graph(dot: Digraph, node: TreeNode, counter: Iterator[int]) -> str:
    node_id = f"n{next(counter)}"

    if node.node_type == "leaf":
        label_lines = [f"Predict: {node.prediction}"]
        if node.raw_prediction and node.prediction != node.raw_prediction:
            label_lines.append(f"(class id: {node.raw_prediction})")
        if node.probabilities:
            label_lines.append("")
            label_lines.append(node.probabilities)

        dot.node(
            node_id,
            "\n".join(label_lines),
            shape="box",
            style="filled",
            fillcolor="#f1f8ff",
        )
        return node_id

    dot.node(node_id, node.condition, shape="ellipse", fontsize="10")

    if node.true_child is not None:
        true_child_id = add_nodes_to_graph(dot, node.true_child, counter)
        dot.edge(node_id, true_child_id, label="True")

    if node.false_child is not None:
        false_child_id = add_nodes_to_graph(dot, node.false_child, counter)
        edge_label = "False"
        if node.else_condition:
            edge_label = f"False\n{node.else_condition}"
        dot.edge(node_id, false_child_id, label=edge_label)

    return node_id


def build_graphviz(model: DecisionTreeClassificationModel, feature_names: List[str], labels: List[str]) -> Digraph:
    root = parse_debug_tree(model.toDebugString, feature_names, labels)
    if root is None:
        raise RuntimeError("Failed to parse the decision tree output.")

    dot = Digraph("pyspark_decision_tree", format="png")
    dot.attr(rankdir="TB", nodesep="0.4", splines="true")

    counter = itertools.count()
    add_nodes_to_graph(dot, root, counter)
    return dot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="outputs/iris_decision_tree",
        help="Output path (without extension) for the rendered Graphviz image.",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Graphviz output format.",
    )
    args = parser.parse_args()

    spark = SparkSession.builder.appName("PySpark multi-class decision tree viz").getOrCreate()

    try:
        df = load_iris_dataset(spark)
        pipeline = build_pipeline(df)
        model = pipeline.fit(df)
        tree_model = model.stages[-1]
        label_indexer: StringIndexer = model.stages[0]
        labels = list(label_indexer.labels)
        feature_columns = [col for col in df.columns if col != "label_str"]

        dot = build_graphviz(tree_model, feature_columns, labels)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered_path = dot.render(str(output_path), format=args.format, cleanup=True)
        print(f"Saved decision tree visualization to {rendered_path}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
