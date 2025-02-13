import argparse
import numpy as np
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from src.agent.agent import PPOAgent
from src.agent.trainer import RLTrainer
from src.env import MarketEnv


def config_reshape(action, size, low, high):
    action = low + (high - low) * action / size
    action = np.clip(action, low, high)
    return action


def train(args):
    def training_function(config, report=None, checkpoint=False):
        n_levels = 10
        env = MarketEnv(n_levels=n_levels, spread_mean=args.spread, volatility=args.volatility)

        precision = int(1e3)
        action_reshape = lambda action: config_reshape(action, precision, env.action_space.low,
                                                       env.action_space.high / 2000)
        state_dim = env.observation_space.shape[0]
        action_dim = [int(precision) for _ in range(env.action_space.shape[0])]
        agent = PPOAgent(
            num_features=state_dim - 4 * n_levels,
            num_depth=n_levels,
            action_dim=action_dim,
            policy_hidden_dims=(state_dim * 24, state_dim * 10, state_dim * 4),
            value_hidden_dims=(state_dim * 10, state_dim * 5, state_dim // 2),
            action_reshape=action_reshape,
            attention_heads=8,
            gamma=config["gamma"],
            eps_clip=config["epsilon"],
            gae_lambda=config["lambd"],
            entropy_coef=config["entropy"],
            lr=[config["lr_policy"], config["lr_value"]],
            batch_size=config["batch_size"]
        )

        trainer = RLTrainer(env, agent)

        if args.pretrained:
            trainer.load()

        trainer.train(num_episodes=int(args.num_episodes), report=report, checkpoint=checkpoint)

    if args.tune:
        config = {
            "gamma": tune.choice([0.9, 0.95, 0.99]),
            "epsilon": tune.uniform(0.1, 0.3),
            "lambd": tune.uniform(0.8, 1.0),
            "entropy": tune.loguniform(1e-5, 1e-2),
            "lr_policy": tune.loguniform(1e-5, 1e-3),
            "lr_value": tune.loguniform(1e-5, 1e-3),
            "batch_size": tune.choice([128, 256, 512])
        }

        algo = OptunaSearch()

        trainable = tune.with_resources(lambda conf: training_function(conf, tune.report),
                                        resources={"gpu": 1})

        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                metric="reward",
                mode="max",
                search_alg=algo,
            ),
            run_config=tune.RunConfig(
                stop={"training_iteration": 100},
            ),
            param_space=config,
        )

        results = tuner.fit()
        print("Best Hyperparameters Found:", results.get_best_result().config)
    else:
        training_function(vars(args), checkpoint=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a PPO agent in a market environment.")

    parser.add_argument('--spread', type=float, default=0.1, help="Mean spread of the market.")
    parser.add_argument('--volatility', type=float, default=0.6, help="Volatility of the market.")

    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor for rewards.")
    parser.add_argument('--epsilon', type=float, default=0.2, help="Clipping parameter for PPO.")
    parser.add_argument('--lambd', type=float, default=0.95, help="Lambda for Generalized Advantage Estimation.")
    parser.add_argument('--entropy', type=float, default=0.1, help="Entropy coefficient for exploration.")

    parser.add_argument('--num_episodes', type=int, default=int(1e4), help="Number of episodes to train.")
    parser.add_argument('--lr_policy', type=float, default=3e-4, help="Learning rate for policy network.")
    parser.add_argument('--lr_value', type=float, default=3e-4, help="Learning rate for value network.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training.")

    parser.add_argument("--pretrained", action="store_true", help="Fine-tune the model.")
    parser.add_argument("--tune", action="store_true",
                        help="Enable hyperparameter optimization using Ray Tune and Optuna.")

    args = parser.parse_args()
    train(args)
