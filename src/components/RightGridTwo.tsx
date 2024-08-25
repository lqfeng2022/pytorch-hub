import { Box, Image, SimpleGrid, Stack, Text, Flex } from '@chakra-ui/react'
import Section from '../entities/Section'

interface Props {
  section: Section
}

const RightGridTwo = ({ section }: Props) => {
  return (
    <Box py={3}>
      <Text as='b' color='gray.600'>{section.name}</Text>
      <SimpleGrid columns={[2, null, 2]} spacing='20px' py={3}>
        <Stack my={2} spacing={2}>
          {section.content.map((p) => <Text key={p.value}>{p.value}</Text>)}
        </Stack>
        <Box maxW="200px">
          <Image src={section.image}/>
        </Box>
      </SimpleGrid>
    </Box>
  )
}

export default RightGridTwo