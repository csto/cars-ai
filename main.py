import time
import argparse
import json
import torch
import transformers

from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast

print("start")

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--content", help="Content")

content = parser.parse_args().content

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

# model = BloomForCausalLM.from_pretrained("mrm8488/bloom-1b3-8bit").to(device)
# tokenizer = BloomTokenizerFast.from_pretrained("mrm8488/bloom-1b3-8bit")

# model = BloomForCausalLM.from_pretrained("bigscience/bloom-7b1").to(device)
# tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-7b1")

# model = BloomForCausalLM.from_pretrained("bigscience/bloom-3b").to(device)
# tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-3b")

# model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b7").to(device)
# tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b7")

# model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b1").to(device)
# tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b1")

model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m").to(device)
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")

prompt = f"""With the following vehicle listing data:
```
{content}
```

The vehicle details from the data are:"""

queries = {
  'vin': 'VIN',
  'name': 'Full year, make, model and trim',
  # 'year': 'Model Year',
  # 'make': 'Manufacturer',
  # 'model': 'Model',
  # 'trim': 'Trim',
  # 'msrp': 'MSRP',
  # 'price': 'Retail Price',
  # 'mileage': 'Mileage',
  # 'engine': 'Engine/Motor or N/A',
  # 'fuel_type': 'Fuel type or N/A',
  # 'transmission': 'Transmission or N/A',
  # 'body_style': 'Body Type/Style or N/A',
  # 'drivetrain': 'Drivetrain (FWD, RWD, AWD, 4x4, N/A)',
  # 'exterior_color': 'Exterior Color or N/A',
  # 'interior_color': 'Interior Color or N/A',
  # 'condition': 'Condition (new, certified, used)',
  # 'title': 'Title (clean/salvage/rebuilt)',
  # 'phone_number': 'Sales Phone Number',
  # 'address': 'Vehicle Location Street, City, State and Zipcode',
  # 'modified': 'Is vehicle modified?',
  # 'features': 'Features',
}

results = {}

for key, value in queries.items():
  new_prompt = prompt + f"\n{value}:"

  inputs = tokenizer(new_prompt, return_tensors="pt").to(device)
  length = len(inputs["input_ids"][0])

  response = tokenizer.decode(model.generate(inputs["input_ids"],
                       max_length=length + 12
                      )[0])

  results[key] = response.replace(new_prompt, "").splitlines()[0].strip()
  prompt = new_prompt + f" {results[key]}"

end = time.time()
print("done", end - start)

# print results as json
print(json.dumps(results))




