#!/usr/env python
import string
import re

import numpy as np
import pandas as pd
from rantanplan import get_scansion
from rantanplan.structures import STRUCTURES
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text


def export_py_code(tree, feature_names, max_depth=100, spacing=4):
    if spacing < 2:
        raise ValueError('spacing must be > 1')
    # Clean up feature names (for correctness)
    nums = string.digits
    alnums = string.ascii_letters + nums
    clean = lambda s: ''.join(c if c in alnums else '_' for c in s)
    features = [clean(x) for x in feature_names]
    features = ['_'+x if x[0] in nums else x for x in features if x]
    if len(set(features)) != len(feature_names):
        raise ValueError('invalid feature names')
    # First: export tree to text
    res = export_text(tree, feature_names=features,
                        max_depth=max_depth,
                        decimals=6,
                        spacing=spacing-1)
    # Second: generate Python code from the text
    skip, dash = ' '*spacing, '-'*(spacing-1)
    code = 'def decision_tree({}):\n'.format(', '.join(features))
    for line in repr(tree).split('\n'):
        code += skip + "# " + line + '\n'
    for line in res.split('\n'):
        line = line.rstrip().replace('|',' ')
        if '<' in line or '>' in line:
            line, val = line.rsplit(maxsplit=1)
            line = line.replace(' ' + dash, 'if')
            line = '{} {:g}:'.format(line, float(val))
        else:
            line = line.replace(' {} class:'.format(dash), 'return')
        code += skip + line + '\n'
    return code


def pad(l, size, element=0):
    return l + [0] * (size - len(l))


def features(scansion, size=100):
    pattern = [s.get("rhyme", 0) for s in scansion]
    min_lengths = [
        s["rhythm"].get("length_range", {}).get("min_length", s["rhythm"]["length"])
        for s in scansion
    ]
    max_lengths = [
        s["rhythm"].get("length_range", {}).get("max_length", s["rhythm"]["length"])
        for s in scansion
    ]
    rhyme_types = [s.get("rhyme_type") for s in scansion]
    if "assonant" in rhyme_types:
        rhyme_type = "assonant"
    elif "consonant" in rhyme_types:
        rhyme_type = "consonant"
    else:
        rhyme_type = "unrhymed"
    structure = scansion[0].get("structure", "unknown")
    return (
        [structure, rhyme_type]
        + pad(pattern, size, element="")
        + pad(min_lengths, size) + pad(max_lengths, size)
    )


def features_regex(scansion, size=100):
    pattern = "".join([s.get("rhyme", "") for s in scansion])
    rhyme_types = [s.get("rhyme_type") for s in scansion]
    if "assonant" in rhyme_types:
        rhyme_type = "assonant"
    elif "consonant" in rhyme_types:
        rhyme_type = "consonant"
    else:
        rhyme_type = "unrhymed"
    length_ranges = [
        range(
            line["rhythm"].get("length_range", {}).get("min_length", line["rhythm"]["length"]),
            line["rhythm"].get("length_range", {}).get("max_length", line["rhythm"]["length"]) + 1)
        for line in scansion]
    regex_features = []
    lengths_features = []
    for _, name, regex, lengths_func in STRUCTURES:
        if callable(regex):
            regex_features.append(int(regex(pattern)))
        else:
            regex_features.append(
                int(bool(re.compile(regex, re.VERBOSE).fullmatch(pattern)))
            )
        lengths_features.append(int(lengths_func(length_ranges)))
    structure = scansion[0].get("structure", "unknown")
    return [structure, rhyme_type] + regex_features + lengths_features


def main(use_regex=False):
    stanzas = pd.read_csv(
        open("evaluation-final.tsv"),
        sep="\t", names=["text", "label"], header=0
    )
    stanzas.label = stanzas.label.apply(
        lambda x: "unknown" if x == "None" else x
    )
    if not use_regex:
        feats = stanzas.text.apply(
            lambda x: features(get_scansion(x, rhyme_analysis=True))
        )
        feats_names = (
            ["rhyme_type"]
            + [f"rhyming_pattern_verse_{i}" for i in range(1, 101)]
            + [f"min_length_verse_{i}" for i in range(1, 101)]
            + [f"max_length_verse_{i}" for i in range(1, 101)]
        )
    else:
        feats = stanzas.text.apply(
            lambda x: features_regex(get_scansion(x, rhyme_analysis=True))
        )
        feats_names = (
            ["rhyme_type"]
            + [f"{name}_regex" for _, name, *_ in STRUCTURES]
            + [f"{name}_lengths" for _, name, *_ in STRUCTURES]
        )

    feats_df = pd.DataFrame(
        np.vstack(feats),
        columns=["predicted"] + feats_names)
    stanzas_feats = pd.concat([stanzas, feats_df], axis=1, ignore_index=True)
    stanzas_feats.columns = stanzas.columns.tolist() + feats_df.columns.tolist()
    stanzas_feats.to_csv("stanzas_feats.csv", header=False)
    train = stanzas_feats[stanzas_feats.label == stanzas_feats.predicted]
    test = stanzas_feats[stanzas_feats.label != stanzas_feats.predicted]

    X_train = train.drop(["text", "label", "predicted"], axis=1)
    y_train = train["label"]

    encoder_columns = ["rhyme_type"]
    if not use_regex:
        encoder_columns += [f"rhyming_pattern_verse_{i}" for i in range(1, 101)]

    encoders = {col: LabelEncoder().fit(stanzas_feats[col])
                for col in encoder_columns}
    for column in encoder_columns:
        X_train[column] = encoders[column].transform(X_train[column])

    clf = DecisionTreeClassifier().fit(X_train, y_train)

    X_test = test.drop(["text", "label", "predicted"], axis=1)
    y_test = test["label"]
    for column in encoder_columns:
        X_test[column] = encoders[column].transform(X_test[column])

    print(export_text(clf, feature_names=feats_names, max_depth=1e3))
    [print(k, e.classes_) for k, e in encoders.items()]
    print(clf.score(X_train, y_train))
    print(classification_report(y_test, clf.predict(X_test)))


if __name__ == "__main__":
    main()
