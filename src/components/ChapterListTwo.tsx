import { Box, List, ListItem, SimpleGrid, Stack, Text } from '@chakra-ui/react'
import Lists from '../components/Lists'
import Content from '../entities/content'
import SectionItem from '../entities/sectionItem'

interface Props {
  leftItems: SectionItem[],
  rightItems: SectionItem[],
  descrip: Content[]
}
const ChapterListTwo = ({leftItems, rightItems, descrip}: Props) => {
  return (
    <>
      <Box py={4} my={5} bg='red.50'>
        <SimpleGrid columns={{ sm: 1, md: 2 }} >
          <Box px={6} py={2} alignItems='center'>
            <List spacing={4}>
              {leftItems.slice(0, 7).map((list) => <ListItem key={list.id}>
                <Lists name={list.name}/>
              </ListItem>)}
            </List>
          </Box>
          <Box px={6} py={2} alignItems='center'>
            <List spacing={4}>
              {rightItems.slice(-6).map((list) => <ListItem key={list.id}>
                <Lists name={list.name}/>
              </ListItem>)}
            </List>
          </Box>
        </SimpleGrid>
      </Box>
      <Stack spacing={4}>
        {descrip.map((p) => <Text key={p.id}>{p.value}</Text>)}
      </Stack>
    </>
  )
}

export default ChapterListTwo