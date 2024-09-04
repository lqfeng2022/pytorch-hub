import { Box, List, ListItem, SimpleGrid, Stack, Text } from '@chakra-ui/react'
import Lists from './Lists'
import Content from '../entities/Content'
import SectionItem from '../entities/SectionItem'


interface Props {
  items: SectionItem[],
  descrip: Content[]
}

const ChapterList = ({ items, descrip }: Props) => {
  return (
    <Box>
      <SimpleGrid spacing={4} columns={{ sm: 1, md: 2 }} >
        <Box 
          p={6} 
          minH='300px' maxH='400px'
          bg='red.50' alignItems='center' // Ensure centered vertically
        >
          <List spacing={4}>
            {items.map((list) => <ListItem key={list.id}>
              <Lists name={list.name}/>
            </ListItem>)}
          </List>
        </Box>
        <Stack spacing={4}>
          {descrip.map((p) => <Text key={p.id}>{p.value}</Text>)}
        </Stack>
      </SimpleGrid>
    </Box>
  )
}

export default ChapterList