import { Box, HStack, Image, Stack, Text } from '@chakra-ui/react'
import Section from '../entities/Section'

interface Props {
  section: Section
}

const BaseGrid = ({ section }: Props) => {

  return (
    <Box py={3}>
      <Stack spacing={3}>
        <Text as='b' fontSize='lg'>{section.name}</Text>
        {section.value && <Text as='i' fontSize='lg'>{section.value}</Text>}
      </Stack>
      <Image py={5} src={section.image}/>
      <Stack spacing={4}>
        {section.content.map((p) => 
          <div key={p.id}>
            {p.title && 
              <HStack>
                <Text>{p.id}</Text>
                <Text as='b'>{p.title}</Text>
              </HStack>}
            <Text>{p.value}</Text>
          </div>
        )}
      </Stack>
    </Box>
  )
}

export default BaseGrid