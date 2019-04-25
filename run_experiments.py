from utils.instrument import VariantGenerator, variant, IO
import os
from datetime import datetime
import shutil
import glob
import paramiko
import time
import utils.ssh as ssh
class VG(VariantGenerator):
    @variant
    def env_name(self):
        return [ 'Reacher-v2' , 'Pusher-v2', 'Thrower-v2','Striker-v2','HalfCheetah-v2','Hopper-v2','Swimmer-v2','Walker2d-v2','Ant-v2','Humanoid-v2']  # 'CellrobotEnv-v0' , 'Cellrobot2Env-v0', 'CellrobotSnakeEnv-v0'  , 'CellrobotSnake2Env-v0','CellrobotButterflyEnv-v0', 'CellrobotBigdog2Env-v0'

    @variant
    def seed(self):
        return [123] #,175,288,1000

    @variant
    def num_steps(self):
        return [2048  ]

    @variant
    def learning_rate(self):
        return [3e-4]

    @variant
    def entropy_coef(self):
        return [0]

    @variant
    def value_loss_coef(self):
        return [0.5]

    @variant
    def ppo_epoch(self):
        return [10]

    @variant
    def num_mini_batch(self):
        return [32]

    @variant
    def gamma(self):
        return [0.99]

    @variant
    def lambda_tau(self):
        return [0.95]

    @variant
    def num_env_steps(self):
        return [1e7]

    ##----------------------------------------------------

algo ='ppo'


exp_id = 1
EXP_NAME ='_{}_RL'.format(algo)
group_note ="************ABOUT THIS EXPERIMENT****************\n" \
            "  " \
        " "


ssh_FLAG = False
AWS_logpath = '/home/drl/PycharmProjects/rl_baselines/pytorch-a2c-ppo-acktr/log-files/AWS_logfiles/'
n_cpu =  8
num_threads = 32


# print choices
variants = VG().variants()
num=0
for v in variants:
    num +=1
    print('exp{}: '.format(num), v)

# save gourp parms
exp_group_dir = datetime.now().strftime("%b_%d")+EXP_NAME+'_Exp{}'.format(exp_id)
group_dir = os.path.join('log-files', exp_group_dir)
os.makedirs(group_dir)

variants = VG().variants()
num = 0
param_dict = {}
for v in variants:
    num += 1
    print('exp{}: '.format(num), v)
    parm = v
    parm = dict(parm, **v)
    param_d = {'exp{}'.format(num): parm}
    param_dict.update(param_d)


IO('log-files/' + exp_group_dir + '/exp_id{}_param.pkl'.format(exp_id)).to_pickle(param_dict)
print(' Parameters is saved : exp_id{}_param.pkl'.format(exp_id))
# save args prameters
with open(group_dir + '/readme.txt', 'wt') as f:
    print("Welcome to Jerry's lab\n", file=f)
    print(group_note, file=f)


log_interval = 1
save_model_interval = 20

full_output = True
evaluate_monitor = False


# SSH Config
if ssh_FLAG:
    hostname = '2402:f000:6:3801:15f4:4e92:4b69:87da' #'2600:1f16:e7a:a088:805d:16d6:f387:62e5'
    username = 'drl'
    key_path = '/home/ubuntu/.ssh/id_rsa_dl'
    port = 22

# run
num_exp =0
for v in variants:
    num_exp += 1
    print(v)
    time_now = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
    # load parm
    exp_name = 'No_{}_{}_PPO-{}'.format(num_exp, v['env_name'], time_now)
    log_dir =  os.path.join(group_dir,exp_name)
    save_dir =  os.path.join(log_dir,'model')
    seed = v['seed']
    env_name = v['env_name']


    gamma = v['gamma']
    lambda_tau = v['lambda_tau']

    num_steps = v['num_steps']
    learning_rate = v['learning_rate']
    entropy_coef = v['entropy_coef']
    value_loss_coef = v['value_loss_coef']
    ppo_epoch = v['ppo_epoch']
    num_env_steps = int(v['num_env_steps'])
    num_mini_batch = v['num_mini_batch']




    os.system("python3 main.py "  +

              " --env-name " + str(env_name) +
              " --algo " + str(algo) +
              " --use-gae "  +
              " --log-interval " + str(log_interval) +
              " --num-steps " + str(num_steps) +
              " --num-processes " + str(n_cpu) +
              " --lr " + str(learning_rate) +
              " --entropy-coef " + str(entropy_coef) +
              " --value-loss-coef " + str(value_loss_coef) +
              " --ppo-epoch " + str(ppo_epoch) +
              " --num-mini-batch " + str(num_mini_batch) +
              " --num-env-steps " + str(num_env_steps) +


              " --gamma " + str(gamma) +
              " --gae-lambda " + str(lambda_tau) +
              " --use-linear-lr-decay " +
              " --use-proper-time-limits " +

              " --save-interval " + str(save_model_interval) +

              " --log-dir " + str(log_dir) +
              " --save-dir " + str(save_dir)

              )


    print("python3 main.py "  +

              " --env-name " + str(env_name) +
              " --algo " + str(algo) +
              " --use-gae "  +
              " --log-interval " + str(log_interval) +
              " --num-steps " + str(num_steps) +
              " --num-processes " + str(n_cpu) +
              " --lr " + str(learning_rate) +
              " --entropy-coef " + str(entropy_coef) +
              " --value-loss-coef " + str(value_loss_coef) +
              " --ppo-epoch " + str(ppo_epoch) +
              " --num-mini-batch " + str(num_mini_batch) +
              " --num-env-steps " + str(num_env_steps) +


              " --gamma " + str(gamma) +
              " --gae-lambda " + str(lambda_tau) +
              " --use-linear-lr-decay " +
                " --use-proper-time-limits " +

              " --save-interval " + str(save_model_interval) +

              " --log-dir " + str(log_dir) +
              " --save-dir " + str(save_dir))
    if ssh_FLAG:
        local_dir = os.path.abspath(group_dir)
        remote_dir = AWS_logpath + exp_group_dir + '/'
        ssh.upload(local_dir, remote_dir, hostname=hostname, port=port, username=username,
                   pkey_path=key_path)


