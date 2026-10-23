def bind_shared_heads(agents):
    if not agents:
        raise ValueError('Central training requires agents')
    root = agents[0]
    for agent in agents[1:]:
        for name in ('actor', 'critic', 'target_actor', 'target_critic', 'actor_optimizer', 'critic_optimizer'):
            setattr(agent, name, getattr(root, name))
        agent.updates = root.updates

def central_update(agents, index, rng):
    agent = agents[index]
    result = agent.update(rng)
    for other in agents:
        other.updates = agent.updates
    return result
