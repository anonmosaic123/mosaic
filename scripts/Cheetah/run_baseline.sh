for seed in 1 2 3 4 5; do
    python train_PEBBLE.py env=mo-halfcheetah-v4 multi_obj=false polite=false weighted=false ensemble_size=4 seed=$seed agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=200 reward_batch=20 reward_update=50 feed_type=$1 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0.1 teacher_eps_skip=0 teacher_eps_equal=0 feed_type=1
done