<img width="1198" height="1008" alt="Screenshot from 2026-07-08 00-23-56" src="https://github.com/user-attachments/assets/438086ac-a213-4be8-b784-b1c05a140da5" />


This repository contains the core Java implementation of the Graph Tsetlin Machine for Foreign Exchange (FX) Regime Anticipation, by Christian Blakely and Melanie Gilmore of La Jolla Private Wealth Group, Wells Fargo Advisors. The architecture is divided into three functional layers: data preprocessing/symbolic embedding, graph message passing, and the logic-based Tsetlin Automata learning execution engine.

## Codebase Architecture Overview

The system is organized into four core packages:

* **`data`**: Handles ingestion and continuous feature calculation over raw hourly market data streams. It directly computes metrics like the Efficiency Ratio (ER) to track signal-to-noise thresholds and the Average True Range (ATR) to establish baseline asset volatility.
* **`embedding`**: Translates the raw indicators and discrete labels into high-dimensional Sparse Binary Hypervectors (SBH) using Vector Symbolic Architecture (VSA). It handles linear scaling to preserve continuous geometric data relationships and implements explicit bitwise `XOR` binding and bundling operations.
* **`graph`**: Constructs and manages the topological relationship mappings between target assets and macro drivers. It coordinates the parallel message-passing layers and updates specialized structural inbox buffers allocated to each neighbor node.
* **`tsetlin`**: Executes the core learning engine by managing decentralized, 2N-state Tsetlin Automata teams. It evaluates deep conjunctive clauses through standard voting, implements Type I and Type II stochastic feedback loops, and symbolically backpropagates structural errors across the graph layout.



<img width="831" height="1210" alt="Screenshot from 2026-07-08 00-25-16" src="https://github.com/user-attachments/assets/5020c246-9a28-4044-ae6d-e86ea28dc130" />

## Disclaimer

The views, thoughts, and code implementations contained in this repository are solely those of the authors and do not represent the official policy, position, or views of Wells Fargo & Company or its affiliates. 

This repository is provided strictly for academic research and educational purposes. The code, models, and backtesting metrics contained herein do not constitute investment advice, financial analysis, or a recommendation to buy or sell any financial instruments or currency pairs.
