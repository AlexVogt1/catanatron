echo "------------------------------------------------------------------------"
echo "Job started on" `date`
echo "------------------------------------------------------------------------"
echo Running on $HOSTNAME...
echo Running on $HOSTNAME... >&2

baseline_experiments=("./experiment_json/baseline/exp_001" "./experiment_json/baseline/exp_002" "./experiment_json/baseline/exp_003" "./experiment_json/baseline/exp_004" "./experiment_json/baseline/exp_005")
switch_experiments=("./experiment_json/switch/exp_009" "./experiment_json/switch/exp_010" "./experiment_json/switch/exp_011" "./experiment_json/switch/exp_012" "./experiment_json/switch/exp_013" "./experiment_json/switch/exp_014")
group_1=("./experiments_cluser/g1s50" "./experiments_cluser/g1s100" "./experiments_cluser/g1s150" "./experiments_cluser/g1s200" "./experiments_cluser/g1s250" "./experiments_cluser/g1s300" "./experiments_cluser/g1s350" "./experiments_cluser/g1s400" "./experiments_cluser/g1s450" "./experiments_cluser/g1s500" "./experiments_cluser/g1s1000")
group_2=("./experiments_cluser/g2s50" "./experiments_cluser/g2s100" "./experiments_cluser/g2s150" "./experiments_cluser/g2s200" "./experiments_cluser/g2s250" "./experiments_cluser/g2s300" "./experiments_cluser/g2s350" "./experiments_cluser/g2s400" "./experiments_cluser/g2s450" "./experiments_cluser/g2s500" "./experiments_cluser/g2s1000")
group_3=("./experiments_cluser/g3s50" "./experiments_cluser/g3s100" "./experiments_cluser/g3s150" "./experiments_cluser/g3s200" "./experiments_cluser/g3s250" "./experiments_cluser/g3s300" "./experiments_cluser/g3s350" "./experiments_cluser/g3s400" "./experiments_cluser/g3s450" "./experiments_cluser/g3s500" "./experiments_cluser/g3s1000")

# source ~/.bashrc
# cd ~/catan/catanatron
# conda activate catanatron

# for i in "${analysis_play_styles[@]}" 
# do
#     echo "$i"
#     python train_ppo.py --env=switch-hard --play_style="$i" --reward_scheme="$i" --exp_type=switch_analysis_killer --action_type=policy --action_space_type=discrete --obs_type=distance --algo=PPO --base_path=./play_style_models/base/
# done
# for i in "${baseline_experiments[@]}" 
# do
#     echo "$i"
#     python train_baseline.py --json_path="$i" 
# done
# for i in "${group_3[@]}" 
# do
#     echo "$i"
#     python my_shap.py --json_path="$i" 
# done

python train_switch.py --json_path="./experiment_json/switch/exp_020"
python train_switch.py --json_path="./experiment_json/switch/exp_019"
# python train_switch.py --json_path="./experiment_json/switch/exp_017"
# python train_switch.py --json_path="./experiment_json/switch/exp_018"

echo "------------------------------------------------------------------------"
echo "Job ended on" `date`
echo "------------------------------------------------------------------------"