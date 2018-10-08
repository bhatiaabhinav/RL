echo "starting for art poisson"
$GYM_PYTHON -um gym_ERSLE.test --env=pyERSEnv-ca-dynamic-1440-v6 --episodes=100 --alloc="[2,2,1,1,1,1,2,0,1,2,1,0,1,1,2,1,2,1,0,1,2,3,2,1,1]"
echo "done for art poisson"

echo "starting for sg poisson"
$GYM_PYTHON -um gym_ERSLE.test --env=SgERSEnv-ca-dynamic-1440-v6 --episodes=100 --alloc="[1,0,1,0,0,0,0,3,0,0,2,3,2,4,2,0,1,2,3,1,0,3,3,1,0]"
echo "done for sg poisson"

echo "starting for art poisson + blips"
$GYM_PYTHON -um gym_ERSLE.test --env=pyERSEnv-ca-dynamic-blips-1440-v6 --episodes=100 --alloc="[2,1,0,1,0,1,2,1,1,2,1,1,1,1,1,2,4,1,1,1,1,3,2,0,1]"
echo "done for art poisson + blips"

echo "starting for sg poisson + blips"
$GYM_PYTHON -um gym_ERSLE.test --env=SgERSEnv-ca-dynamic-blips-1440-v6 --episodes=100 --alloc="[0,0,0,0,0,0,0,3,0,0,1,4,3,4,1,1,2,2,4,1,0,2,4,0,0]"
echo "done for sg poisson + blips"