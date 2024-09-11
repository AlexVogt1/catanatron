echo "------------------------------------------------------------------------"
echo "Job started on" `date`
echo "------------------------------------------------------------------------"
echo Running on $HOSTNAME...
echo Running on $HOSTNAME... >&2

baseline_experiments=("./experiment_json/baseline/exp_001" "./experiment_json/baseline/exp_002" "./experiment_json/baseline/exp_003" "./experiment_json/baseline/exp_004" "./experiment_json/baseline/exp_005")
switch_experiments=("./experiment_json/switch/exp_009" "./experiment_json/switch/exp_010" "./experiment_json/switch/exp_011" "./experiment_json/switch/exp_012" "./experiment_json/switch/exp_013" "./experiment_json/switch/exp_014")
# source ~/.bashrc
# cd ~/src/EDPCGARL/gym-pcgrl

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
for i in "${switch_experiments[@]}" 
do
    echo "$i"
    python train_switch.py --json_path="$i" 
done

echo "------------------------------------------------------------------------"
echo "Job ended on" `date`
echo "------------------------------------------------------------------------"