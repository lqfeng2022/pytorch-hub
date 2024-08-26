import { Box, Container, Grid, GridItem, Show } from '@chakra-ui/react';
import ChapterList from './components/ChapterList';
import Cover from './components/Cover';

function HomePage() {
  return (
    <Container maxW='1200px' px='10px' mt='72px'>
      <Grid
        templateAreas={{ base: `'main'`, lg: `'aside main'` }}
        templateColumns={{ base: '1fr', lg: '350px 1fr' }}
        gap={4}
        >
        <Show above='lg'>
          <GridItem area='aside' px={3} py={1} bg='gray.100' >
            <Box
              py={3}
              position='sticky'
              top='72px'
              maxH='calc(100vh - 72px)'
              overflowY='auto'
            >
              <ChapterList/>
            </Box>
          </GridItem>
        </Show>
        <GridItem area='main'>
          <Cover/>
        </GridItem>
      </Grid>
    </Container>
  );
}

export default HomePage;