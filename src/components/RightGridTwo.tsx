import { Box, Image, SimpleGrid, Stack, Text } from '@chakra-ui/react'
import Section from '../entities/section'

const RightGridTwo = ({ section }: { section: Section }) => {
  return (
    <Box pt={5}>
      <Stack spacing={3}>
        <Text as='b' fontSize='lg' color='gray.600'>{section.name}</Text>
        {section.value && <Text as='i' fontSize='lg'>{section.value}</Text>}
      </Stack>
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