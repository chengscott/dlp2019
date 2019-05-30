from deep_rl import *
import sys


# DQN
def dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.0005)
    config.network_fn = lambda: VanillaNet(
        config.action_dim, FCBody(config.state_dim, hidden_units=(32, )))
    config.replay_fn = lambda: Replay(memory_size=int(5e4), batch_size=128)
    #config.replay_fn = lambda: AsyncReplay(memory_size=int(5e4), batch_size=128)

    config.random_action_prob = LinearSchedule(1.0, 0.0066, 1e4)
    config.discount = 0.95
    config.target_network_update_freq = 50
    config.exploration_steps = 1000
    config.double_q = False
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.eval_interval = 100
    config.max_steps = 7e4
    config.async_actor = False
    run_steps(DQNAgent(config))


# DDPG
def ddpg_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(1e6)
    #config.eval_interval = int(1e4)
    #config.eval_episodes = 20

    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim,
        config.action_dim,
        actor_body=FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body=TwoLayerFCBodyWithAction(
            config.state_dim, config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: Replay(memory_size=int(1e4), batch_size=64)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim, ), std=LinearSchedule(0.2))
    config.warm_up = int(1e4)
    config.target_network_mix = 1e-3  # tau
    run_steps(DDPGAgent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    select_device(-1)
    # select_device(0)

    assert (len(sys.argv) > 1), 'Required option: `-c` or `-p`'
    arg = sys.argv[1]
    assert (arg in ('-c', '-p')), 'Option must be: `-c` or `-p`'
    if arg == '-c':
        game = 'CartPole-v0'
        dqn_feature(game=game)
    elif arg == '-p':
        game = 'Pendulum-v0'
        ddpg_continuous(game=game)
