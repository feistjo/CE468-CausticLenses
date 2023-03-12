Based on https://github.com/MattFerraro/causticsEngineering

To run,
[jfs6711@ras CE468-CausticLenses]$ source ~hardav/cuda-env.csh
[jfs6711@ras CE468-CausticLenses]$ cd labs/src/caustic/
[jfs6711@ras caustic]$ make
[jfs6711@ras caustic]$ time ../../bin/linux/release/causticlens
0.553u 0.215s 0:00.94 80.8%     0+0k 176+31256io 1pf+0w
[jfs6711@ras caustic]$ 

To run the single-threaded version, follow the instructions at https://github.com/MattFerraro/causticsEngineering