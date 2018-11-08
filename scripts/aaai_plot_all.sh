style='seaborn'

# BSS real:
sh plot_pdf.sh BSSEnv-v0 --title='Bike-Sharing-Real' --logdir=$1 --style=$style --metrics='Reward' --scales='-200:0' --run_ids='RTrailer(capacity=5):rtrailer_cap5,DDPG-CP:sddpg_rmsac_wolpert,DDPG-CS:cddpg_rmsac_wolpert'
sh plot_pdf.sh BSSEnv-v1 --title='Bike-Sharing-Poisson-Origin' --logdir=$1 --style=$style --metrics='Reward' --scales='-75:0' --run_ids='RTrailer(capacity=5):rtrailer_cap5,DDPG-CP:sddpg_rmsac_wolpert,DDPG-CS:cddpg_rmsac_wolpert'
sh plot_pdf.sh BSSEnv-v2 --title='Bike-Sharing-Poisson-Origin-Dest' --logdir=$1 --style=$style --metrics='Reward' --scales='-75:0' --run_ids='RTrailer(capacity=5):rtrailer_cap5,DDPG-CP:sddpg_rmsac_wolpert,DDPG-CS:cddpg_rmsac_wolpert'

# Artifical non blips: 
sh plot_pdf.sh pyERSEnv-ca-dynamic-cap6-30-v6 --title='ERS-Art.-Possion(Local-cons.)' --logdir=$1 --style=$style --metrics='Reward' --scales='150:300' --run_ids='Greedy-Static:greedy,DDPG-CP:sddpg_rmsac_wolpert,DDPG-CS:cddpg_rmsac_wolpert'
sh plot_pdf.sh pyERSEnv-ca-dynamic-constraints-30-v6 --title='ERS-Art.-Poisson(Regional-cons.)' --logdir=$1 --style=$style --metrics='Reward' --scales='150:300' --run_ids='Greedy-Static:greedy,DDPG-CP:sddpg_rmsac_wolpert,DDPG-CS:cddpg_rmsac_wolpert'

# Singapore non blips:
sh plot_pdf.sh SgERSEnv-ca-dynamic-cap6-30-v6 --title='ERS-Sg.-Possion(Local-cons.)' --logdir=$1 --style=$style --metrics='Reward' --scales='210:360' --run_ids='Greedy-Static:greedy,DDPG-CP:sddpg_rmsac_wolpert,DDPG-CS:cddpg_rmsac_wolpert'
sh plot_pdf.sh SgERSEnv-ca-dynamic-constraints-30-v6 --title='ERS-Sg.-Possion(Regional-cons.)' --logdir=$1 --style=$style --metrics='Reward' --scales='210:360' --run_ids='Greedy-Static:greedy,DDPG-CP:sddpg_rmsac_wolpert,DDPG-CS:cddpg_rmsac_wolpert'

#Artificial blips:
sh plot_pdf.sh pyERSEnv-ca-dynamic-blips-cap6-30-v6 --title='ERS-Art.-Possion-Surge(Local-cons.)' --logdir=$1 --smoothing=500 --style=$style --metrics='Reward:Reward,Surge-Reward:Blip_Reward' --scales='150:300,0:70' --run_ids='Greedy-Static:greedy,DDPG-CP:sddpg_rmsac_wolpert,DDPG-CS:cddpg_rmsac_wolpert'
sh plot_pdf.sh pyERSEnv-ca-dynamic-blips-constraints-30-v6 --title='ERS-Art.-Possion-Surge(Regional-cons.)' --logdir=$1 --smoothing=500 --style=$style --metrics='Reward:Reward,Surge-Reward:Blip_Reward' --scales='150:300,0:70' --run_ids='Greedy-Static:greedy,DDPG-CP:sddpg_rmsac_wolpert,DDPG-CS:cddpg_rmsac_wolpert'

#Singapore blips:
sh plot_pdf.sh SgERSEnv-ca-dynamic-blips-cap6-30-v6 --title='ERS-Sg.-Possion-Surge(Local-cons.)' --logdir=$1 --smoothing=500 --style=$style --metrics='Reward:Reward,Surge-Reward:Blip_Reward' --scales='210:360,0:70' --run_ids='Greedy-Static:greedy,DDPG-CP:sddpg_rmsac_wolpert,DDPG-CS:cddpg_rmsac_wolpert'
sh plot_pdf.sh SgERSEnv-ca-dynamic-blips-constraints-30-v6 --title='ERS-Sg.-Possion-Surge(Regional-cons.)' --logdir=$1 --smoothing=500 --style=$style --metrics='Reward:Reward,Surge-Reward:Blip_Reward' --scales='210:360,0:70' --run_ids='Greedy-Static:greedy,DDPG-CP:sddpg_rmsac_wolpert,DDPG-CS:cddpg_rmsac_wolpert'

cd $1

graphs_dir=./../graphs
rm -r $graphs_dir
mkdir -p $graphs_dir

cp -r BSSEnv-v0/plots $graphs_dir/BikeSharing-real
cp -r BSSEnv-v1/plots $graphs_dir/BikeSharing-poisson-origin
cp -r BSSEnv-v2/plots $graphs_dir/BikeSharing-poisson-origin-dest
cp -r pyERSEnv-ca-dynamic-cap6-30-v6/plots $graphs_dir/ERS-art-poisson-capacity6
cp -r pyERSEnv-ca-dynamic-constraints-30-v6/plots $graphs_dir/ERS-art-poisson-nestedconstraints
cp -r SgERSEnv-ca-dynamic-cap6-30-v6/plots $graphs_dir/ERS-Sg-poisson-capacity6
cp -r SgERSEnv-ca-dynamic-constraints-30-v6/plots $graphs_dir/ERS-Sg-poisson-nestedconstraints
cp -r pyERSEnv-ca-dynamic-blips-cap6-30-v6/plots $graphs_dir/ERS-art-poisson+blips-capacity6
cp -r pyERSEnv-ca-dynamic-blips-constraints-30-v6/plots $graphs_dir/ERS-art-poisson+blips-nestedconstraints
cp -r SgERSEnv-ca-dynamic-blips-cap6-30-v6/plots $graphs_dir/ERS-Sg-poisson+blips-capacity6
cp -r SgERSEnv-ca-dynamic-blips-constraints-30-v6/plots $graphs_dir/ERS-Sg-poisson+blips-nestedconstraints