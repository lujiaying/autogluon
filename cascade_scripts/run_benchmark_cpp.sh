# ======== How to train and save models using CPP datasets ==========
# === Run one session with default exp arguments ===
# python cascade_scripts/benchmark_cpp.py \
#     --exp_result_save_dir ExpResults/CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022 \
#     --session_names 4d20c284-0cd9-4889-8d34-cee136f906bc

# === Run three sessions with specified fit_infer_limit arguments ===
# python cascade_scripts/benchmark_cpp.py \
#     --fit_infer_limit 1e-3 \
#     --exp_result_save_dir ExpResults/CPP-Benchmark-n4dg.2xlarge-4h8c-Sep282022-fit_infer_limit1ms \
#     --session_names 060016f7-0d1d-41a4-83e0-cd1eb254b140 eb920d43-2d72-4744-852e-2a5deb8cdfc0 f29d388f-4f4a-4160-99cb-6aef2f5be7f1

# === Run all 60 session with default exp arguments ===
# python cascade_scripts/benchmark_cpp.py \
#     --exp_result_save_dir ExpResults/CPP-Benchmark-n4dg.2xlarge-4h8c-All60sessions \
#     --session_start_end 0 59



# ======== How to execute cascade algorithms using saved models ==========
# === one specific session ===
# cascade_result_fname indicates that the experiment result stores in
# ${cpp_result_dir}/${session_names}/scores/${cascade_result_fname}
python -m cascade_scripts.exec_fit_cascade_post_cpp \
    --cpp_result_dir ExpResults/CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022 \
    --cascade_result_fname cascade_results.csv \
    --session_names 3564a7a7-0e7c-470f-8f9e-5a029be8e616 \
    --infer_limit_list None 1e-2 3e-3
# === all sessions in the `cpp_result_dir` ===
# python -m cascade_scripts.exec_fit_cascade_post_cpp \
#     --cpp_result_dir ExpResults/CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022 \
#     --cascade_result_fname cascade_results.csv \
#     --infer_limit_list None 1e-2 3e-3


# ======== How to generate figures using cascade results ==========
# python -m cascade_results.generate_cpp_report