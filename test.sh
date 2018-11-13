module load python_gpu/3.6.4 &&
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.001 --upscale_factor 2 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.01 &&
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.001 --upscale_factor 2 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.001 &&
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.001 --upscale_factor 2 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.0001 &&
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.001 --upscale_factor 4 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.01 &&
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.001 --upscale_factor 4 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.001 &&
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.001 --upscale_factor 4 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.0001 &&
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.001 --upscale_factor 8 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.01 &&
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.001 --upscale_factor 8 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.001 &&
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.001 --upscale_factor 8 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.0001 &&
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.0003 --upscale_factor 2 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.01 &&
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.0003 --upscale_factor 2 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.001 &&
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.0003 --upscale_factor 2 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.0001 &&
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.0003 --upscale_factor 4 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.01 &&
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.0003 --upscale_factor 4 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.001 &&
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.0003 --upscale_factor 4 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.0001 &&
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.0003 --upscale_factor 8 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.01 &&
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.0003 --upscale_factor 8 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.001 &&
bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.0003 --upscale_factor 8 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.0001

bsub -W 10:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=3072,ngpus_excl_p=1]" python main.py --lr 0.001 --upscale_factor 2 --batch_size 200 --epochs 100 --n_save 2 --gamma 0.01

