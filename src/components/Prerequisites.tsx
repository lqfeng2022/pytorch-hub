import { Box, Stack, Text, Image } from '@chakra-ui/react'
import prere_img from '../assets/prerequisites.jpeg'

const Prerequisites = () => {
  const prere = {
    title: "7. Prerequisites",
    subtitle: "What should we do before this trainig course, first of all, you should know Python programming languages. Then you gotta know some mathematic knowledges, like Matrix computation, calculus, probability and statistics.",
    image: prere_img,
    python: "PyTorch is a Python-based framework, so if you wanna learn PyTorch, first of all, you gotta know a bit of Python programming language, like variables, functions, classes, data-type, data-structure. Python is pretty simple, if you're familiar with other programming language like Java or JavaScript, you can mast it quickly.", 
    maths: [
      { id: 1, 
        title: "Linear Algebra",
        value: "Understanding vectors, matrices, and operations like matrix multiplication is crucial since neural networks and other ML algorithms rely heavily on these concepts. I will cover it in the next chapter, called Tensors."
      },
      { 
        id: 2, 
        title: "Calculus",
        value: "Be familiar with derivative and integral is pretty useful, especially for understanding how backpropagation and gradient descent work in training process of neural networks."
      },
      { id: 3, 
        title: "Probability and Statistics",
        value: "Be familiar with probability, random variables, probability distribution and normal distribution. It will help you understand concepts like loss functions, activation functions, and the overall behavior of models."
      },
    ]
  }

  return (
    <Box py={5}>
      <Text as='b' fontSize='lg'>{prere.title}</Text>
      <Text fontSize='lg' pt={3}>{prere.subtitle}</Text>
      <Image py={5} src={prere.image}/>
      <Stack spacing={4}>
        <Box>
          <Text as='b'>PyThon Programming Language</Text>
          <Text py={2}>{prere.python}</Text>
        </Box>
        <Box>
          <Text as='b'>Mathematical Fondations</Text>
          {prere.maths.map((p) => <Text py={2} key={p.id}>{p.value}</Text>)}
        </Box>
      </Stack>
    </Box>
  )
}

export default Prerequisites