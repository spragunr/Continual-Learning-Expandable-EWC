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
    
    perm = np.arange(len(indices))
    np.random.shuffle(perm)

    return indices, perm

def apply_permutation(image, indices, perm):

    permute_sample = image[indices]

    print("PS: \n {}".format(permute_sample))
    
    permute_sample = permute_sample[perm]
    
    print("PS_shuff: \n {}".format(permute_sample))

    image[indices] = permute_sample
    
    return image

images = [np.array([1, 5, 7, 4, 6, 3, 8, 12, 14, 15]), np.array([1, 4, 2, 5, 3, 6, 7, 9, 14, 0])]

print(images)

indices, perm = generate_percent_permutation(0.3, len(images[0]))

print("INDICES: \n {}".format(indices))
print("PERM: \n {}".format(perm))

for i in range(len(images)):
    images[i] = apply_permutation(images[i], indices, perm)

print(images)



