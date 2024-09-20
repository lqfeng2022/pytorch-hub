import { Box, HStack, Image, Stack, Text } from '@chakra-ui/react'
import Section from '../entities/section'

const BaseGrid = ({ section }: { section: Section }) => {

  return (
    <Box pt={5}>
      <Stack spacing={3}>
        <Text as='b' fontSize='lg' color='gray.600'>{section.name}</Text>
        {section.value && <Text as='i' fontSize='lg'>{section.value}</Text>}
      </Stack>
      <Image py={5} src={section.image}/>
      <Stack spacing={4}>
        {section.content.map((p) => 
          <div key={p.id}>
            {p.title && 
              <>
                <HStack>
                  <Text>{p.id}</Text>
                  <Text as='b'>{p.title}</Text>
                </HStack>
                <Text pl={4}>{p.value}</Text>
              </>
            }
            {!p.title && <Text>{p.value}</Text>}
          </div>
        )}
      </Stack>
    </Box>
  )
}

export default BaseGrid