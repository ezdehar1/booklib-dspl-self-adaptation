\# BookLib DSPL Self-Adaptation

BookLib is a microservice-based research case study designed to support the evaluation of application-level self-adaptation using Dynamic Software Product Lines (DSPL). The system models a book browsing and purchasing application with explicit variability, including optional and alternative microservice variants such as display, inventory, recommender, review, and advertisement services. This variability enables the generation of valid system configurations and creates clear trade-offs between response time (RT) and user experience (UX).

This repository provides the replication package for experiments on DSPL-guided self-adaptation using Deep Reinforcement Learning. It includes the BookLib implementation, feature-model/configuration artifacts, workload and dataset materials, predictive-model components, DDQN decision-making code, and GA baseline artifacts. The package is intended to support reproducibility and further research on self-adaptive microservice systems, application-level configuration adaptation, and QoS--UX trade-off analysis.

\#\# Repository Structure

\- \`BookLib Manifest/\`: Kubernetes manifests and deployment-related files for BookLib.  
\- \`Data/\`: Workload, dataset, and experimental data files.  
\- \`DDQN\_Code\_GA/\`: DDQN implementation and GA baseline code.  
\- \`PPM-Models/\`: Predictive performance model artifacts.

\#\# Citation

If you use this artifact, please cite the associated paper:

Self-Adaptive Microservice Systems Using Dynamic Software Product Lines and Deep Reinforcement Learning.  
