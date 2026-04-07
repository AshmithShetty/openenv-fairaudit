import asyncio
import websockets
import json

async def run_evaluation():
    uri = "ws://localhost:7860/ws/eval_agent_001"
    
    print("Connecting to FairAudit Environment...")
    async with websockets.connect(uri) as ws:
        
        async def send_command(payload):
            await ws.send(json.dumps(payload))
            response = await ws.recv()
            return json.loads(response)

        
        # TASK 1: DATASET SCAN 
        
        print("\n--- STARTING TASK 1: DATASET SCAN ---")
        obs = await send_command({"command": "reset", "task_name": "dataset-scan"})
        print(f"Initialized. Columns: {obs['data']['dataset_info']['columns']}")
        
        print("\nAction 1: Inspect 'zip_code'")
        res = await send_command({
            "command": "step",
            "action": {"action_type": "inspect_column", "parameters": {"column": "zip_code"}, "reasoning": "Check structure."}
        })
        print(f"Reward: {res['reward']:.2f} | Feedback: {res['info']['reward_details']['feedback']}")
        
        print("\nAction 2: Flag Bias Proxy")
        res = await send_command({
            "command": "step",
            "action": {"action_type": "flag_bias", "parameters": {"column": "zip_code", "reason": "Redlining proxy"}, "reasoning": "Identify proxy."}
        })
        print(f"Reward: {res['reward']:.2f} | Feedback: {res['info']['reward_details']['feedback']}")

        print("\nAction 3: Submit Task 1")
        res = await send_command({
            "command": "step",
            "action": {"action_type": "submit", "parameters": {}, "reasoning": "Done."}
        })
        print(f"Final T1 Score: {res['reward']:.2f} | Done: {res['done']}")

   
        # TASK 2: MODEL AUDIT 
      
        print("\n--- STARTING TASK 2: MODEL AUDIT ---")
        await send_command({"command": "reset", "task_name": "model-audit"})
        
        print("\nAction 1: Compute Disparate Impact")
        res = await send_command({
            "command": "step",
            "action": {
                "action_type": "compute_metric", 
                "parameters": {"protected_attribute": "race", "target_column": "loan_approved", "privileged_value": 0, "positive_outcome": 1}, 
                "reasoning": "Calculate DI."
            }
        })
        print(f"Metrics Computed | Reward: {res['reward']:.2f}")

        print("\nAction 2: Flag Violation")
        res = await send_command({
            "command": "step",
            "action": {"action_type": "flag_violation", "parameters": {"protected_attribute": "race", "severity": "HIGH"}, "reasoning": "Severe DI."}
        })
        print(f"Reward: {res['reward']:.2f} | Feedback: {res['info']['reward_details']['feedback']}")

        print("\nAction 3: Submit Task 2")
        res = await send_command({
            "command": "step",
            "action": {"action_type": "submit", "parameters": {}, "reasoning": "Done."}
        })
        print(f"Final T2 Score: {res['reward']:.2f} | Done: {res['done']}")

      
        # TASK 3: BIAS MITIGATION 
      
        print("\n--- STARTING TASK 3: BIAS MITIGATION ---")
        await send_command({"command": "reset", "task_name": "bias-mitigation"})
        
        print("\nAction 1: Evaluate Baseline Model")
        res = await send_command({
            "command": "step",
            "action": {
                "action_type": "evaluate_model", 
                "parameters": {"feature_columns": ["annual_income", "debt_to_income_ratio", "fico_score", "employment_length"], "target_column": "loan_approved"}, 
                "reasoning": "Get baseline."
            }
        })
        acc = res['observation']['last_action_result'].get('test_accuracy', 0)
        print(f"Baseline Accuracy: {acc:.2f} | Reward: {res['reward']:.2f}")

        print("\nAction 2: Apply Mitigation (Drop DTI - The Pareto Trap)")
        res = await send_command({
            "command": "step",
            "action": {
                "action_type": "apply_mitigation", 
                "parameters": {"strategy": "drop_feature", "feature": "debt_to_income_ratio"}, 
                "reasoning": "Drop highly predictive feature."
            }
        })
        print(f"Reward: {res['reward']:.2f} | Feedback: Feature dropped.")

        print("\nAction 3: Re-Evaluate Model (Triggering the Crash)")
        res = await send_command({
            "command": "step",
            "action": {
                "action_type": "evaluate_model", 
                "parameters": {"feature_columns": ["annual_income", "fico_score", "employment_length"], "target_column": "loan_approved"}, 
                "reasoning": "Re-evaluate without DTI."
            }
        })
        new_acc = res['observation']['last_action_result'].get('test_accuracy', 0)
        print(f"New Accuracy: {new_acc:.2f} | Reward: {res['reward']:.2f} | Feedback: {res['info']['reward_details']['feedback']}")

        print("\nAction 4: Verify Fairness Post-Mitigation")
        res = await send_command({
            "command": "step",
            "action": {
                "action_type": "verify_fairness", 
                "parameters": {"protected_attribute": "race", "privileged_value": 0, "positive_outcome": 1}, 
                "reasoning": "Check DI improvement."
            }
        })
        print(f"Reward: {res['reward']:.2f} | Feedback: {res['info']['reward_details']['feedback']}")

        print("\nAction 5: Submit Task 3")
        res = await send_command({
            "command": "step",
            "action": {"action_type": "submit", "parameters": {}, "reasoning": "Done."}
        })
        print(f"Final T3 Score: {res['reward']:.2f} | Done: {res['done']}")
        print("\n--- END-TO-END EVALUATION COMPLETE ---")

if __name__ == "__main__":
    asyncio.run(run_evaluation())