LEONHARD = True
# ./test2.sh > out 2>&1 &
lrs = [0.001]
upscale_factor = [2, 4, 8]
gammas = [0.001]
thetas = [0.01]
#ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

cnt = 0
for lr in lrs:
	for up in upscale_factor:
		for g in gammas:
			for theta in thetas:
				#for r in ratio:
					if cnt == 0:
						print('module load python_gpu/3.6.4 &&')
					print('bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py '
					      '--lr {} --upscale_factor {} --batch_size 100 --epochs 100 --n_save 2 --gamma {} --theta {} &&'
					      .format(lr, up, g, theta))

					cnt += 1
					if cnt % 14 == 0:
						print('sleep 18000 &&')
						print('module load python_gpu/3.6.4 &&')




# select[gpu_model1==GeForceGTX1080Ti]