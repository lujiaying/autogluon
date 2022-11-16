# Handbook for scripts under dir `cascade_scripts`

```
.
|- benchmark_cpp.py: script used to run autogluon with different fit arguments on cpp datasets
|- exec_fit_cascade_post_cpp.py: script used to run cascade algorithms after getting saved ag model files
|- run_benchmark_cpp.sh: real examples about how to run `benchmark_cpp.py` and `exec_fit_cascade_post_cpp.py`
|- generate_cpp_report.py: how to draw figures (tradeoffs between accuracy and infer speed) using cascade results.
|- example_for_covertype_dataset.py: a runnable example that show POC of cascade algorithm (no need to modify anything).

```

Under `if __name__ == '__main__':`, I've put the arguments help documents. If you have any questions, feel free to contact me `jiaying.lu@emory.edu`.