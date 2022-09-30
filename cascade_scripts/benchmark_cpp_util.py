import os
# get remaining session names
# all session names
all_cpp_sessions = set(os.listdir('datasets/cpp_research_corpora/2021_60datasets'))
print('all sessions cnt:', len(all_cpp_sessions))
# get finished session names
finished_cpp_sessions = set(os.listdir('ExpResults/CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022'))
print('finished sessions cnt', len(finished_cpp_sessions))
# get remaining
remaining_cpp_sessions = all_cpp_sessions - finished_cpp_sessions
print('remaining session cnt', len(remaining_cpp_sessions))
print(remaining_cpp_sessions)
# Sep 28
{'060016f7-0d1d-41a4-83e0-cd1eb254b140', 'eb920d43-2d72-4744-852e-2a5deb8cdfc0', 'f29d388f-4f4a-4160-99cb-6aef2f5be7f1', 'e549baf7-41b6-43a7-9fce-fb742908c18f', 
'd33a68e4-23f2-4734-8e37-116b63258c15', 'fa4789d9-a5a0-4c83-9bcf-9b7d8ebdb410', 'f7840a3c-02d4-49e8-a73d-fff30a06d6f7', 
'e522248d-8a23-4f2a-8a95-d41a9a777304', 'dfe28a8f-dfc3-4e4b-a580-0415aef4fb5c', 'c80d326c-3cc8-4b50-b8c6-d67901dcfdcb', 
'ddc65c81-a9b7-4516-a939-821a0401943d', 'e4367988-17e8-4727-ac84-58603b9931e4', 'fc10f6bb-320b-42f9-8dbb-bc589b894cce', 
'f749589f-e747-41b7-b5ef-dcddf54fdad5', 'fde2fa26-4262-4e51-b914-d9724fc925c2', 'f50f35f6-67d7-4c3c-ae31-7d4bced81bf5'}