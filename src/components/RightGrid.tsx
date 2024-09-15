import { Box, Image, Stack, Text, Flex, HStack, Grid } from '@chakra-ui/react'
import Section from '../entities/section'

const RightGrid = ({ section }: { section: Section }) => {
  return (
    <Box pt={5}>
      <Stack spacing={3}>
        <Text as='b' fontSize='lg' color='gray.600'>{section.name}</Text>
        {section.value && <Text as='i' fontSize='lg'>{section.value}</Text>}
      </Stack>
      <Grid templateColumns={{ base: '1fr', md: '2fr 3fr' }} mt={5} gap={5}>
        <Box order={[2, null, 1]}>
          <Stack spacing={2}>
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
        </Box>
        <Box order={[1, null, 2]}>
          <Flex alignItems='center' justifyContent='center'>
            <Image src={section.image}/>
          </Flex>
        </Box>
      </Grid>
    </Box>
  )
}

export default RightGrid