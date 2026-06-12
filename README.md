# HGX
HGX (Hierarchical Guided eXtraction) is a neurosymbolic framework that combines Large Language Models and Answer Set Programming through hierarchical fact extraction. This repository contains the implementation of HGX, and the needed files for testing and benchmarking.

HGX implements a hierarchical parent-child organization of predicates that enables structural pruning of irrelevant branches, and guarantees schema-constrained LLM outputs based on the type specifications under each predicate in an application file.

The benchmarks that can be tested are defined by the application files, these are E (Electrical Grid Domain) and G+LNRS (Layered Graph + Labyrinth, Nomystery, Ricochet Robots, and Sokoban). LNRS can also be tested alone, without G.

There's two dataset entries, one for E and another for G+LNRS, the first has 33 prompts, while the other has a total of 188 prompts that can be tested.

It requires a .env file that details the LLM key to enable the connection with the server housing the LLMs.

The dictionary can be modified to manage different results, while the normalize_group(name) function can be easily modified to manage problem groupings.
