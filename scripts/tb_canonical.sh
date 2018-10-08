cd $OPENAI_LOGDIR
$ACTIVATE_GYM_PYTHON
tensorboard --logdir=\
capv6:pyERSEnv-ca-dynamic-cap6-30-v6,\
blipscapv6:pyERSEnv-ca-dynamic-blips-cap6-30-v6,\
sgcapv6:SgERSEnv-ca-dynamic-cap6-30-v6,\
sgblipscapv6:SgERSEnv-ca-dynamic-blips-cap6-30-v6,\
ncv6:pyERSEnv-ca-dynamic-constraints-30-v6,\
blipsncv6:pyERSEnv-ca-dynamic-blips-constraints-30-v6,\
sgncv6:SgERSEnv-ca-dynamic-constraints-30-v6, \
sgblipsncv6:SgERSEnv-ca-dynamic-blips-constraints-30-v6, \
bssv0:BSSEnv-v0,\
bssv1:BSSEnv-v1,\
bssv2:BSSEnv-v2\
 $1 $2 $3 $4 $5
