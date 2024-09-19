import { Box, List, ListItem, Stack, Text } from '@chakra-ui/react'
import Content from '../entities/content'
import SectionItem from '../entities/SectionItem-temp'
import Lists from './Lists'

interface Props {
  items: SectionItem[],
  descrip: Content[]
}
const ChapterList = ({ items, descrip }: Props) => {
  return (
    <Box py={5}>
      <Box p={6} mb={5} bg='red.50'>
        <List spacing={4}>
          {items.map((list) => <ListItem key={list.id}>
            <Lists name={list.name}/>
          </ListItem>)}
        </List>
      </Box>
      <Stack spacing={4}>
        {descrip.map((p) => <Text key={p.id}>{p.value}</Text>)}
      </Stack>
    </Box>
  )
}

export default ChapterList