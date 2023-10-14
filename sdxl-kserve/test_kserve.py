import requests
import json
import numpy as np

from PIL import Image

input = {
	"instances": [
		{
			"data": "a futuristic cat learning from an ai"
		}
	]
}

headers= {
	"Host": "torchserve-default.example.com"
}

url = "http://ac845061f828240dab04f85abc63eb7e-fa9f4e13d0b94c70.elb.ap-south-1.amazonaws.com:80/v1/models/sdxl:predict"

response = requests.post(url, data=json.dumps(input), headers=headers)

# with open("raw.txt", "w") as f:
# 	f.write(response.text)

# with open("raw.txt", "r") as f:
# 	a = f.read()

image = Image.fromarray(np.array(json.loads(response.text)['predictions'][0], dtype="uint8"))
image.save("out.jpg")