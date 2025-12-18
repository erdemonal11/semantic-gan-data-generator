# Schema Guided RDF Triple Generation

## Overview

This repository presents an experimental analysis of RDF triple generation under an explicit ontology schema.
The schema is derived from standard Semantic Web vocabularies such as FOAF, BIBO, W3C ORG and W3C TIME.

## Scope

The system generates individual RDF statements of the form.

```turtle
:subject  :predicate  :object .
```

Generation is restricted to the triple level.
This allows direct measurement of schema violations and avoids effects introduced by graph assembly or reasoning.
The output is a stream of candidate RDF statements rather than a completed knowledge graph.

## Evaluation

Generated triples are analyzed empirically with respect to schema adherence.
Evaluation considers aggregate behavior across runs, including consistency rates, relation distributions, and cardinality effects.

Schema consistency is treated as an observed property of the generated data.
It is not enforced as a guaranteed outcome.

## Relation to [TOntoGen](https://corescholar.libraries.wright.edu/knoesis/977)

[TOntoGen](https://corescholar.libraries.wright.edu/knoesis/977) controls ontology population through explicit parameters.
In contrast, this study does not prescribe target class or relation frequencies.

The contribution is analytical rather than infrastructural.
It examines how closely generated triples align with schema expectations when the schema is used as a guiding reference rather than a strict generator.
