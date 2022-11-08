# Sep 18
# python cascade_scripts/benchmark_cpp.py \
#     --exp_result_save_dir ExpResults/CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022 \
#     --session_names 4d20c284-0cd9-4889-8d34-cee136f906bc

# Sep 28
# 0.0007s = Feature Preprocessing Time (1 row | 10000 batch size)
#                Feature Preprocessing requires 66.19% of the overall inference constraint (0.001s)
#                0.0003s inference time budget remaining for models...
# python cascade_scripts/benchmark_cpp.py \
#     --fit_infer_limit 1e-3 \
#     --exp_result_save_dir ExpResults/CPP-Benchmark-n4dg.2xlarge-4h8c-Sep282022-fit_infer_limit1ms \
#     --session_names 060016f7-0d1d-41a4-83e0-cd1eb254b140 eb920d43-2d72-4744-852e-2a5deb8cdfc0 f29d388f-4f4a-4160-99cb-6aef2f5be7f1

# Sep 29
# python -m cascade_scripts.exec_fit_cascade_post_cpp \
#     --cpp_result_dir ExpResults/CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022

# Oct 2
# python cascade_scripts/benchmark_cpp.py \
#     --fit_infer_limit 5e-3 \
#     --exp_result_save_dir ExpResults/CPP-Benchmark-n4dg.2xlarge-4h8c-Sep282022-fit_infer_limit5ms \
#     --session_start_end 0 4

# Oct 6
python -m cascade_scripts.exec_fit_cascade_post_cpp \
    --cpp_result_dir ExpResults/CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022 \
    --cascade_result_fname cascade_results_spec_less_thresholds.csv \
    --session_names 3564a7a7-0e7c-470f-8f9e-5a029be8e616 \
    --infer_limit_list 1.0 1e-2 5e-3 3e-3