# Given input is a list of tensor strings with values. The goal is to extract just the numerical values from these strings.

input_tensors = [
    "tensor(0.0320, device='cuda:0')", "tensor(0.3090, device='cuda:0')", "tensor(0.6840, device='cuda:0')",
    "tensor(0.3740, device='cuda:0')", "tensor(0.7390, device='cuda:0')", "tensor(0.3920, device='cuda:0')",
    "tensor(0.8150, device='cuda:0')", "tensor(0.7480, device='cuda:0')", "tensor(0.7250, device='cuda:0')",
    "tensor(0.7870, device='cuda:0')", "tensor(0.7110, device='cuda:0')", "tensor(0.9460, device='cuda:0')",
    "tensor(0.8730, device='cuda:0')", "tensor(0.8440, device='cuda:0')", "tensor(0.9030, device='cuda:0')",
    "tensor(0.8240, device='cuda:0')", "tensor(0.9860, device='cuda:0')", "tensor(0.9230, device='cuda:0')",
    "tensor(0.9200, device='cuda:0')", "tensor(0.9080, device='cuda:0')", "tensor(0.1600, device='cuda:0')",
    "tensor(0.4150, device='cuda:0')", "tensor(0.2580, device='cuda:0')", "tensor(0.2320, device='cuda:0')",
    "tensor(0.1990, device='cuda:0')", "tensor(0.2100, device='cuda:0')", "tensor(0.1860, device='cuda:0')",
    "tensor(0.4780, device='cuda:0')", "tensor(0.6130, device='cuda:0')", "tensor(0.6600, device='cuda:0')",
    "tensor(0.7580, device='cuda:0')", "tensor(0.8520, device='cuda:0')", "tensor(0.9250, device='cuda:0')",
    "tensor(0.0430, device='cuda:0')", "tensor(0.2910, device='cuda:0')"
]

# Extracting numerical values using list comprehension
extracted_values = [float(tensor.split('(')[1].split(',')[0]) for tensor in input_tensors]

print(extracted_values)
