import { Box, Stack, Text, Image } from '@chakra-ui/react'
import ptfeatures_img from '../assets/ptfeatures_img.jpeg'

const PTFeatures = () => {
  const ptFeatures = {
    title: "6.1 PyTorch Features",
    image: ptfeatures_img,
    values: [
      { id: 1, 
        title: "Dynamic Computational Graphs",
        value: "PyTorch uses dynamic computational graphs. This means that the graph is built on the fly as operations are performed, allowing for more flexibility in model design and debugging. You can modify the graph during runtime, making it easier to work with complex models."
      },
      { 
        id: 2, 
        title: "Tensor Operations",
        value: "PyTorch supports multi-dimensional arrays (called tensors), similar to NumPy, but with the added capability of running on GPUs. This makes it suitable for large-scale computations needed for deep learning."
      },
      { id: 3, 
        title: "GPU Acceleration",
        value: "PyTorch is designed to take full advantage of GPUs for accelerated computation. By leveraging CUDA (NVIDIA's parallel computing platform), PyTorch allows for fast and efficient training of deep learning models."
      },
      { id: 4, 
        title: "Pre-built Models and Libraries",
        value: "PyTorch has a vast ecosystem, including libraries like TorchVision (for image processing), TorchText (for NLP), and TorchAudio (for audio processing). It also offers pre-built models through tools like torch.hub and torchvision.models."
      },
      { id: 5, 
        title: "Integration with Python",
        value: "PyTorch is fully integrated with Python, making it easy to use with other Python libraries like NumPy, SciPy, and Pandas. This makes it a preferred choice for many developers and researchers who are already familiar with Python."
      },
    ]
  }

  return (
    <Box py={5}>
      <Text as='b' fontSize='lg'>{ptFeatures.title}</Text>
      <Image py={5} src={ptFeatures.image}/>
      <Stack spacing={4}>
        {ptFeatures.values.map((p) => 
          <div key={p.id}>
            <Text as='b'>{p.title}</Text>
            <Text py={2}>{p.value}</Text>
          </div>
        )}
      </Stack>
    </Box>
  )
}

export default PTFeatures