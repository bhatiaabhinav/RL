cd $WORKSPACE
cd baselines
cd scripts

echo "testing sddpg_rmsac_wolpert algo for all envs"

cd ers
sh create_test.sh variants/sddpg_rmsac_wolpert.sh

for env_id in pyERSEnv-ca-dynamic-cap6-30-v6 pyERSEnv-ca-dynamic-constraints-30-v6 SgERSEnv-ca-dynamic-cap6-30-v6 SgERSEnv-ca-dynamic-constraints-30-v6
do
    echo "starting for $env_id"
    sh test.sh $env_id sddpg_rmsac_wolpert
    echo "done for $env_id$"
done

for env_id in pyERSEnv-ca-dynamic-blips-cap6-30-v6 pyERSEnv-ca-dynamic-blips-constraints-30-v6 SgERSEnv-ca-dynamic-blips-cap6-30-v6 SgERSEnv-ca-dynamic-blips-constraints-30-v6
do
    echo "starting for $env_id"
    sh test.sh $env_id sddpg_rmsac_wolpert
    echo "done for $env_id$"
done

cd ./..
cd bss
sh create_test.sh variants/sddpg_rmsac_wolpert.sh

for env_id in v0 v1 v2
do
    echo "starting for $env_id"
    sh test.sh $env_id sddpg_rmsac_wolpert
    echo "done for $env_id$"
done

echo "done for everything"