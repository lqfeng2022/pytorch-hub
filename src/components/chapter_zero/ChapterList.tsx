import { Box, List, ListItem, SimpleGrid, Stack, Text } from '@chakra-ui/react'
import covers from '../../data/covers'
import Lists from '../Lists'

const ChapterList = () => {
  const { content: lists, description: descript } = covers[0].list

  return (
    <Box>
      <SimpleGrid 
        minH='350px' 
        p={6} 
        my={3} 
        spacing={4} 
        columns={{ sm: 1, md: 2 }} 
        bg='red.50' alignItems='center' // Ensure centered vertically
      >
        <List spacing={4}>
          {lists.slice(0, 4).map((list) => <ListItem key={list.id}> 
            <Lists name={list.name}/>
          </ListItem>)}
        </List>
        <List spacing={4}>
          {lists.slice(-4).map((list) => <ListItem key={list.id}>
            <Lists name={list.name}/>
          </ListItem>)}
        </List>
      </SimpleGrid>
      <Stack my={5} spacing={4}>
        {descript.map((p) => <Text key={p.id}>{p.value}</Text>)}
      </Stack>
    </Box>
  )
}

export default ChapterList