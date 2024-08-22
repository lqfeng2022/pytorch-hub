import { SimpleGrid, List, ListItem, Flex, Icon, Text, Box, Stack } from '@chakra-ui/react'
import { FaCircle } from 'react-icons/fa'

const ChapterList = () => {
  const lists = [
    {id: 1, name: 'Artificial Intelligence'},
    {id: 2, name: 'Machne Learning'},
    {id: 3, name: 'Deep Learning'},
    {id: 4, name: 'Neural Network'},
    {id: 5, name: 'Framework and Libraries'},
    {id: 6, name: 'PyTorch'},
    {id: 7, name: 'Prerequisites'},
    {id: 8, name: 'How to Take this Course'}
  ]

  const introducing = [
    {id: 1, content: "In this chapter, we're gonna dive into some fundamental concepts of Artificial Intelligence and explore a few key AI frameworks and libraries."},
    {id: 2, content: "First off, we gotta get what AI, Machine Learning, Deep Learning, and Neural Networks really are. After that, I'll introduce you to some of the most useful Machine Learning frameworks and libraries. For beginners, PyTorch stands out as a powerful and user-friendly framework, so we'll be using PyTorch to build our next projects."},
    {id: 3, content: "Before we get started, it's helpful to have a basic understanding of Python, such as knowing about variables, functions, classes, data types, and so on. But don't worry if you're not up to speed on Python yet, I've got another course that covers it, and it's pretty straightforward. I'm confident you'll pick it up quickly."},
    {id: 4, content: "As for the math involved, in the upcoming chapters, when we start building models, we'll touch on some mathematical concepts. If you're not familiar with them, or if you've never learned them before, don't stress, I'll break things down visually to make them easier to understand as we go."}
  ]

  return (
    <Box>
      {/* Chapter content lists */}
      <SimpleGrid p={6} my={3} spacing={4} columns={{ sm: 1, md: 2 }} bg='red.50'>
        <List spacing={4}>
          {lists.slice(0, 4).map((list) => <ListItem key={list.id}>
            <Flex align='center'>
              <Icon boxSize='25px' mr={3} as={FaCircle} color='red.300' />
              <Text as='b' fontSize='lg'>{list.name}</Text>
            </Flex>
          </ListItem>)}
        </List>
        <List spacing={4}>
          {lists.slice(-4).map((list) => <ListItem key={list.id}>
            <Flex align='center'>
              <Icon boxSize='25px' mr={3} as={FaCircle} color='red.300' />
              <Text as='b' fontSize='lg'>{list.name}</Text>
            </Flex>
          </ListItem>)}
        </List>
      </SimpleGrid>
      {/* Chapter content introducing */}
      <Stack my={5} spacing={4}>
        {introducing.map((p) => <Text key={p.id}>{p.content}</Text>)}
      </Stack>
    </Box>
  )
}

export default ChapterList