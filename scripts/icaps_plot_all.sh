style='seaborn'

# BSS real:
sh plot_pdf.sh BSSEnv-v0 --title='Bike-Sharing' --logdir=$1 --style=$style --metrics='Reward' --scales='-200:0' --run_ids='RTrailer(capacity=5):rtrailer_cap5,DDPG-CP:ddpg-cp-*,DDPG-CS:ddpg-cs*,DDPG-ApprOpt:ddpg-scoptnet-*'


# Singapore non blips:
sh plot_pdf.sh SgERSEnv-ca-dynamic-cap6-30-v6 --title='ERS-Sg.-Possion' --logdir=$1 --style=$style --metrics='Reward' --scales='250:400' --run_ids='Greedy-Static:greedy,DDPG-CP:ddpg-cp-*,DDPG-CS:ddpg-cs-*,DDPG-ApprOpt:ddpg-scoptnet-*'


# Singapore blips:
sh plot_pdf.sh SgERSEnv-ca-dynamic-blips-cap6-30-v6 --title='ERS-Sg.-Possion+Surge' --logdir=$1 --smoothing=500 --style=$style --metrics='Reward:Reward,Surge-Reward:Blip_Reward' --scales='250:400,0:70' --run_ids='Greedy-Static:greedy,DDPG-CP:ddpg-cp-*,DDPG-CS:ddpg-cs-*,DDPG-ApprOpt:ddpg-scoptnet-*'

cd $1

graphs_dir=./../graphs
rm -r $graphs_dir
mkdir -p $graphs_dir

cp -r BSSEnv-v0/plots $graphs_dir/BikeSharing-real
cp -r SgERSEnv-ca-dynamic-cap6-30-v6/plots $graphs_dir/ERS-Sg-poisson-capacity6
cp -r SgERSEnv-ca-dynamic-blips-cap6-30-v6/plots $graphs_dir/ERS-Sg-poisson+blips-capacity6