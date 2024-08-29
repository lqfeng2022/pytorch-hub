import { Box, Image, SimpleGrid, Stack, Text, Flex } from '@chakra-ui/react'
import Section from '../entities/Section'

const LeftGrid = ({ section }: { section: Section }) => {
  return (
    <Box py={3}>
      <Text as='b' fontSize='lg' color='gray.600'>{section.name}</Text>
      <SimpleGrid columns={[1, null, 2]} spacing='20px' pt={3}>
        <Flex alignItems='center' justifyContent='center'>
          <Image src={section.image}/>
        </Flex>
        <Stack my={2} spacing={2}>
          {section.content.map((p) => <Text key={p.value}>{p.value}</Text>)}
        </Stack>
      </SimpleGrid>
    </Box>
  )
}

export default LeftGrid