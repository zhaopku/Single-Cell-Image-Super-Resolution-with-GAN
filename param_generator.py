LEONHARD = True
# ./test.sh > out 2>&1 &
lrs = [0.001, 0.0003]
upscale_factor = [2, 4]
gammas = [0.001, 0.0001]
thetas = [0.1, 0.001]

cnt = 0
for lr in lrs:
	for up in upscale_factor:
		for g in gammas:
			for theta in thetas:
				if cnt == 0:
					print('module load python_gpu/3.6.4 &&')
				print('bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py '
				      '--lr {} --upscale_factor {} --batch_size 100 --epochs 100 --n_save 2 --gamma {} --theta {} &&'
				      .format(lr, up, g, theta))

				cnt += 1
				if cnt % 16 == 0:
					print('sleep 32400 &&')
					print('module load python_gpu/3.6.4 &&')




# select[gpu_model1==GeForceGTX1080Ti]