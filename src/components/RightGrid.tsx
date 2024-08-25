import { Box, Image, SimpleGrid, Stack, Text, Flex } from '@chakra-ui/react'
import Section from '../entities/Section'

interface Props {
  section: Section
}

const RightGrid = ({ section }: Props) => {
  return (
    <Box py={3}>
      <Text as='b'>{section.name}</Text>
      <SimpleGrid columns={[1, null, 2]} spacing='20px'>
        <Stack my={2} spacing={2}>
          {section.content.map((p) => <Text key={p.value}>{p.value}</Text>)}
        </Stack>
        <Flex alignItems='center' justifyContent='center'>
          <Image src={section.image}/>
        </Flex>
      </SimpleGrid>
    </Box>
  )
}

export default RightGrid