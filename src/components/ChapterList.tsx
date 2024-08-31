import { Box, List, ListItem, SimpleGrid, Stack, Text } from '@chakra-ui/react'
import Lists from './Lists'
import Content from '../entities/Content'
import SectionItem from '../entities/SectionItem'


interface Props {
  litems: SectionItem[],
  ritems: SectionItem[],
  descrip: Content[]
}

const ChapterList = ({ litems, ritems, descrip }: Props) => {
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
          {litems.map((list) => <ListItem key={list.id}> 
            <Lists name={list.name}/>
          </ListItem>)}
        </List>
        <List spacing={4}>
          {ritems.map((list) => <ListItem key={list.id}>
            <Lists name={list.name}/>
          </ListItem>)}
        </List>
      </SimpleGrid>
      <Stack my={5} spacing={4}>
        {descrip.map((p) => <Text key={p.id}>{p.value}</Text>)}
      </Stack>
    </Box>
  )
}

export default ChapterList