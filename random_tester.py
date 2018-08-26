import numpy as np

# length = 20
# percent = .3
# 
# original = np.arange(length)
# 
# print(original)
# 
# perm_size = int(length * percent)
# 
# indices = np.random.choice(length, size=perm_size, replace=False)
# 
# permuted = original[indices]
# np.random.shuffle(permuted)
# 
# original[indices] = permuted
# 
# print(original)


def generate_percent_permutation(percent, length):
    
    perm_size = int(length * percent)
    
    indices = np.random.choice(length, size=perm_size, replace=False)
    
    return indices

def apply_permutation(image, perm):

    permute_sample = image[perm]
    np.random.shuffle(permute_sample)
    
    image[perm] = permute_sample
    
    return image

image = np.arange(784)

print(image)

perm = generate_percent_permutation(0.3, len(image))

image = apply_permutation(image, perm)

print(image)



