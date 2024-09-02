import { Box, Image, SimpleGrid, Stack, Text, Flex, HStack } from '@chakra-ui/react'
import Section from '../entities/Section'

const RightGrid = ({ section }: { section: Section }) => {
  return (
    <Box pt={3} pb={5}>
      <Stack spacing={3}>
        <Text as='b' fontSize='lg' color='gray.600'>{section.name}</Text>
        {section.value && <Text as='i' fontSize='lg'>{section.value}</Text>}
      </Stack>
      <SimpleGrid columns={[1, null, 2]} spacing='20px'>
        <Stack my={2} spacing={2}>
          {/* {section.content.map((p) => <Text key={p.value}>{p.value}</Text>)} */}
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
        <Flex alignItems='center' justifyContent='center'>
          <Image src={section.image}/>
        </Flex>
      </SimpleGrid>
    </Box>
  )
}

export default RightGrid