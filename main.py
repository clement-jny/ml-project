from datasets import load_dataset

# print(load_dataset('squad', split='train')[0])
print(load_dataset('wikipedia', '20220301.fr', split='train', trust_remote_code=True)[0])
# print(load_dataset('wikipedia', language='fr', date='20220301', split='train', trust_remote_code=True)[0])