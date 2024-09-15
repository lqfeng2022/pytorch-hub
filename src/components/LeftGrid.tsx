import { Box, Flex, Grid, HStack, Image, Stack, Text } from '@chakra-ui/react'
import Section from '../entities/section'

const LeftGrid = ({ section }: { section: Section }) => {
  return (
    <Box pt={5}>
      <Stack spacing={3}>
        <Text as='b' fontSize='lg' color='gray.600'>{section.name}</Text>
        {section.value && <Text as='i' fontSize='lg'>{section.value}</Text>}
      </Stack>
      <Grid templateColumns={{ base: '1fr', md: '3fr 2fr' }} gap={5} pt={3}>
        <Flex alignItems='center' justifyContent='center'>
          <Image src={section.image}/>
        </Flex>
        <Stack my={2} spacing={2}>
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
      </Grid>
    </Box>
  )
}

export default LeftGrid