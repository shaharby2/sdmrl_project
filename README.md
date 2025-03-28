# **Optimizing Electricity Market Participation Using Hybrid Reinforcement Learning**  
**Shahar Ben Yehuda, Ori Noriani**

## **Project Overview**
This project explores how **Reinforcement Learning (RL) can optimize energy storage and trading decisions** in a dynamic electricity market. We implement a **hybrid learning approach** that combines **expert demonstrations with RL**, allowing agents to learn more efficiently and maximize profits while ensuring household electricity demand is met.

Our work consists of three main stages:
1. **Environment & Agent Implementation**: Simulating an electricity market with battery storage.
2. **Training RL Agents**: Training **PPO, SAC, TD3, DDPG, and A2C** to optimize energy management.
3. **Hybrid Learning with Expert Data**: Pretraining agents using **expert demonstrations** before fine-tuning them with RL.

---

### Directory Structure**
```
├── IM_expert_files/              # Expert data & pretrained policies
│   ├── expert_dataset.pth        # Processed expert dataset
│   ├── pretrained_expert_policy.pth  # Expert policy weights
│   ├── smart_expert_data.csv     # Expert-generated state-action dataset
│
├── electricity_market_zips/      # Trained baseline models (RL-only)
│   ├── ppo_electricity_market.zip
│   ├── sac_electricity_market.zip
│   ├── td3_electricity_market.zip
│   ├── ddpg_electricity_market.zip
│   ├── a2c_electricity_market.zip
│
├── fine_tuned_models/            # Hybrid-trained models (Expert + RL)
│   ├── fine_tuned_ppo.zip
│   ├── fine_tuned_sac.zip
│   ├── fine_tuned_td3.zip
│   ├── fine_tuned_ddpg.zip
│   ├── fine_tuned_a2c.zip
│
├── logs/                         # Training logs (loss, rewards)
│   ├── monitor.csv                # Logging agent performance
│   ├── progress.csv               # RL training metrics
│
├── Code/   
├── electricity_market_env.py      # Market simulation & RL environment
├── training.py                    # RL agent training & evaluation
├── plot_graphs.py                  # Training statistics visualization
├── part3_hybrid_learning.py        # Hybrid training approach (Expert + RL)

```
## **How to Run the Code**  

### **Train RL Agents (Baseline) - Section 2**  
To train and evaluate the RL models (**PPO, SAC, TD3, A2C, DDPG**):  
```bash
python training.py
```
- This will train the agents and save the **baseline models** in `electricity_market_zips/`.  

---

### **Run Hybrid Learning - Section 3**  
To generate expert data and train **hybrid RL models** (pretrained on expert demonstrations):  
```bash
python part3_hybrid_learning.py
```
- This will create the **expert dataset** in `IM_expert_files/`.  
- Fine-tuned **hybrid models** will be saved in `fine_tuned_models/`.  

---

### **Visualize Training Results**  
To plot training reward curves and compare performance across models:  
```bash
python plot_graphs.py
```
- Generates **graphs showing RL training progress and hybrid model improvements**.  

---

## **Key Findings**
- **SAC outperformed other RL algorithms**, balancing profit maximization and demand fulfillment.
- **Hybrid learning (expert + RL) significantly improved training efficiency**, allowing agents to learn faster.
- **Fine-tuned SAC achieved the best trade-off**, demonstrating the effectiveness of hybrid RL in energy trading.

---

## **Future Work**
- **Multi-Agent RL**: Simulating competitive electricity markets.
- **Integrating Forecasting**: Predicting energy demand & prices using LSTMs.
- **Adaptive Learning**: Dynamically adjusting RL rewards based on market conditions.

---


