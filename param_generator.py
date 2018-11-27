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

"""
module load python_gpu/3.6.4
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python test_raw_images.py --in_path=10x_dapi --out_path=10x_dapi_scale --upscale_factor 2 --batch_size 100 --n_save 10
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python test_raw_images.py --in_path=20x_dapi --out_path=20x_dapi_scale --upscale_factor 2 --batch_size 100 --n_save 10
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python test_raw_images.py --in_path=10x_dapi --out_path=10x_dapi_scale --upscale_factor 4 --batch_size 100 --n_save 10
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python test_raw_images.py --in_path=20x_dapi --out_path=20x_dapi_scale --upscale_factor 4 --batch_size 100 --n_save 10
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python test_raw_images.py --in_path=10x_dapi --out_path=10x_dapi_scale --upscale_factor 8 --batch_size 100 --n_save 10
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python test_raw_images.py --in_path=20x_dapi --out_path=20x_dapi_scale --upscale_factor 8 --batch_size 50 --n_save 20

module load python_gpu/3.6.4
bsub -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python test_raw_images.py --in_path=40x_dapi --out_path=40x_dapi_scale --upscale_factor 2 --batch_size 25 --n_save 40
bsub -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python test_raw_images.py --in_path=40x_dapi --out_path=40x_dapi_scale --upscale_factor 4 --batch_size 25 --n_save 40
bsub -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=5072,ngpus_excl_p=1]" python test_raw_images.py --in_path=40x_dapi --out_path=40x_dapi_scale --upscale_factor 8 --batch_size 25 --n_save 40


module load python_gpu/3.6.4
bsub -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python test_images.py --in_path=40x_dapi --out_path=40x_dapi_simulate --upscale_factor 2 --batch_size 25 --n_save 40
bsub -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python test_images.py --in_path=40x_dapi --out_path=40x_dapi_simulate --upscale_factor 4 --batch_size 25 --n_save 40
bsub -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=5072,ngpus_excl_p=1]" python test_images.py --in_path=40x_dapi --out_path=40x_dapi_simulate --upscale_factor 8 --batch_size 25 --n_save 40

bsub -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python test_images.py --in_path=20x_dapi --out_path=20x_dapi_simulate --upscale_factor 2 --batch_size 25 --n_save 40
bsub -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python test_images.py --in_path=20x_dapi --out_path=20x_dapi_simulate --upscale_factor 4 --batch_size 25 --n_save 40
bsub -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=5072,ngpus_excl_p=1]" python test_images.py --in_path=20x_dapi --out_path=20x_dapi_simulate --upscale_factor 8 --batch_size 25 --n_save 40
bsub -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python test_images.py --in_path=10x_dapi --out_path=10x_dapi_simulate --upscale_factor 2 --batch_size 25 --n_save 40
bsub -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python test_images.py --in_path=10x_dapi --out_path=10x_dapi_simulate --upscale_factor 4 --batch_size 25 --n_save 40
bsub -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=5072,ngpus_excl_p=1]" python test_images.py --in_path=10x_dapi --out_path=10x_dapi_simulate --upscale_factor 8 --batch_size 25 --n_save 40


"""