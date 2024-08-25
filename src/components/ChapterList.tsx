import { HStack, List, ListItem, Stack, Text } from '@chakra-ui/react'

const ChapterList = () => {
  const lists = [
    { id: 0, 
      name: "Artificial Intelligence",
      items: [
        { id: 1, name: "What's AI" },
        { id: 2, name: "Machine Learning" },
        { id: 3, name: "Deep Learning" },
        { id: 4, name: "Neural Network" },
        { id: 5, name: "ML Frameworks and Libraries" },
        { id: 6, name: "PyTorch" },
        { id: 7, name: "Prerequisites" },
        { id: 8, name: "How to take this Course" }
      ]
    },
    { id: 1, 
      name: "TENSORs",
    },
    { id: 2, 
      name: "A Straight Line Model",
    },
    { id: 3,
      name: "A Binary Classification Model"
    },
    { id: 4, 
      name: "A Convolutional Neural Network Model"
    },
    { id: 5, 
      name: "A Vision Transformer Model"
    },
    { id: 6,
      name: "A Language Translation Model"
    },
    { id: 7, 
      name: "A Speech Recognition Model"
    },
    { id: 8, 
      name: "A Speech Generation Model"
    },
    { id: 9, 
      name: "A Recommendation System Model"
    }
  ]
  return (
    <List>
      <ListItem py='6px'>
        <Text fontSize='md'>Introduction</Text>
      </ListItem>
      {lists.map((list) => (
        <ListItem py='6px' key={list.id}>
          <HStack spacing={3}>
            <Text>{list.id}</Text>
            <Text fontSize='md'>{list.name}</Text>
          </HStack>
        </ListItem>
      ))}
      <ListItem py='6px'>
        <Text fontSize='md'>Dedication</Text>
      </ListItem>
      <ListItem py='6px'>
        <Text fontSize='md'>Additional Resources</Text>
      </ListItem>
      <ListItem py='6px'>
        <Text fontSize='md'>Credits</Text>
      </ListItem>
    </List>

  )
}

export default ChapterList